from pathlib import Path

from aws_cdk import Stack, RemovalPolicy, Duration, CfnOutput
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3_deployment
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from aws_cdk import aws_apigateway as apigw
from constructs import Construct

# Repository root: the directory that CONTAINS the `circuitsentry` package.
# Lambda assets must be packaged from here, not from `circuitsentry/` itself,
# because every handler string ("circuitsentry.backend...") and every internal
# import ("from circuitsentry.vision.pipeline import inspect") expects
# `circuitsentry` to be an importable top-level package under /var/task/.
REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = REPO_ROOT / "dashboard"

# Keep non-runtime directories out of the Lambda zip.
LAMBDA_ASSET_EXCLUDE = [
    ".git",
    ".github",
    ".claude",
    "cdk.out",
    "docs",
    "eval",
    "infra",
    "tests",
    "dashboard",
    "**/__pycache__",
    "*.pyc",
    "*.egg-info",
    ".venv",
    "venv",
    "node_modules",
    "*.md",
]

# ---------------------------------------------------------------------------
# TODO (deployment blocker — NOT solved here): the Lambda functions below are
# packaged from source only. Their third-party dependencies (opencv-python-
# headless, numpy; boto3 is present in the Lambda runtime but pinning it is
# safer) are NOT included, so a real `cdk deploy` produces functions that fail
# at import time with ModuleNotFoundError: cv2.
#
# Two supported ways to fix this before deploying:
#   (a) Lambda layer(s) carrying the dependencies. numpy + opencv-python-
#       headless are large (~100MB+ unzipped, near the 250MB Lambda limit), so
#       use opencv-python-headless, not opencv-python. AWS-community "Klayers"
#       publishes a public opencv-python-headless layer per region that can be
#       referenced by ARN via _lambda.LayerVersion.from_layer_version_arn().
#   (b) CDK asset bundling with a Docker image matching the
#       Runtime.PYTHON_3_12 manylinux target, e.g.
#           _lambda.Code.from_asset(str(REPO_ROOT), bundling=BundlingOptions(
#               image=_lambda.Runtime.PYTHON_3_12.bundling_image,
#               command=["bash", "-c", "pip install -r requirements.txt -t /asset-output && cp -r circuitsentry /asset-output"],
#           ))
#       This requires Docker at synth time, which is not available in this
#       project's development sandbox, so it is deliberately left uncommitted
#       and unverified rather than shipped as an untested stub.
# ---------------------------------------------------------------------------


class CircuitSentryStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        lambda_code = _lambda.Code.from_asset(str(REPO_ROOT), exclude=LAMBDA_ASSET_EXCLUDE)

        self.images_bucket = s3.Bucket(
            self, "ImagesBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        self.runs_table = dynamodb.Table(
            self, "RunsTable",
            partition_key=dynamodb.Attribute(name="run_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="seq", type=dynamodb.AttributeType.NUMBER),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.perception_fn = _lambda.Function(
            self, "PerceptionFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="circuitsentry.backend.lambda_perception.handler.lambda_handler",
            code=lambda_code,
            timeout=Duration.seconds(30),
            memory_size=1024,
            environment={"RUNS_TABLE_NAME": self.runs_table.table_name},
        )
        self.images_bucket.grant_read(self.perception_fn)
        self.runs_table.grant_write_data(self.perception_fn)

        perception_task = tasks.LambdaInvoke(
            self, "Perception",
            lambda_function=self.perception_fn,
            output_path="$.Payload",
        )
        # POST /capture starts the execution before the client has finished
        # uploading the frame, so the first Perception invocation can hit
        # NoSuchKey. Retry with backoff to give the upload time to land (and
        # to absorb transient Lambda/service faults generally).
        perception_task.add_retry(
            errors=["States.ALL"],
            interval=Duration.seconds(2),
            max_attempts=4,
            backoff_rate=2.0,
        )

        pass_state = sfn.Pass(self, "Pass")
        flag_state = sfn.Pass(self, "FlagForApproval")
        recapture_state = sfn.Pass(self, "Recapture")
        # spec §8: a repeatedly-failing perception (persistent AlignmentError,
        # a frame that never arrives) must land in a manual-review terminal
        # state rather than failing the execution and silently dropping the board.
        manual_review_state = sfn.Pass(self, "ManualReview")
        perception_task.add_catch(manual_review_state, errors=["States.ALL"])
        # Recapture loops back to Perception once the capture client has fulfilled
        # the re-capture request (event carries the new frame_key on re-entry).
        recapture_state.next(perception_task)

        choice = sfn.Choice(self, "Decide") \
            .when(sfn.Condition.string_equals("$.next_action", "flag_for_approval"), flag_state) \
            .when(sfn.Condition.string_equals("$.next_action", "recapture"), recapture_state) \
            .otherwise(pass_state)

        definition = perception_task.next(choice)

        self.state_machine = sfn.StateMachine(
            self, "InspectionStateMachine",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=Duration.minutes(5),
        )

        capture_fn = _lambda.Function(
            self, "CaptureFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="circuitsentry.backend.lambda_api.capture.lambda_handler",
            code=lambda_code,
            environment={
                "STATE_MACHINE_ARN": self.state_machine.state_machine_arn,
                "IMAGES_BUCKET_NAME": self.images_bucket.bucket_name,
            },
        )
        self.state_machine.grant_start_execution(capture_fn)
        self.images_bucket.grant_read_write(capture_fn)

        runs_fn = _lambda.Function(
            self, "RunsFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="circuitsentry.backend.lambda_api.runs.list_handler",
            code=lambda_code,
            environment={"RUNS_TABLE_NAME": self.runs_table.table_name},
        )
        runs_detail_fn = _lambda.Function(
            self, "RunsDetailFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="circuitsentry.backend.lambda_api.runs.detail_handler",
            code=lambda_code,
            environment={"RUNS_TABLE_NAME": self.runs_table.table_name},
        )
        approve_fn = _lambda.Function(
            self, "ApproveFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="circuitsentry.backend.lambda_api.approve.lambda_handler",
            code=lambda_code,
            environment={"RUNS_TABLE_NAME": self.runs_table.table_name},
        )
        self.runs_table.grant_read_data(runs_fn)
        self.runs_table.grant_read_data(runs_detail_fn)
        self.runs_table.grant_write_data(approve_fn)

        self.api = apigw.RestApi(
            self, "CircuitSentryApi",
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
            ),
        )
        self.api.root.add_resource("capture").add_method("POST", apigw.LambdaIntegration(capture_fn))
        runs_resource = self.api.root.add_resource("runs")
        runs_resource.add_method("GET", apigw.LambdaIntegration(runs_fn))
        run_item = runs_resource.add_resource("{id}")
        run_item.add_method("GET", apigw.LambdaIntegration(runs_detail_fn))
        run_item.add_resource("approve").add_method("POST", apigw.LambdaIntegration(approve_fn))

        # --- Dashboard static site (spec §5) ---------------------------------
        # NOTE for the operator: dashboard/index.html carries a placeholder
        # <meta name="api-base-url"> value. Templating the deployed API URL
        # into it automatically needs a CDK custom resource / build step and is
        # out of scope here — after the first deploy, set that meta tag to the
        # ApiUrl output below and redeploy (BucketDeployment re-uploads).
        self.dashboard_bucket = s3.Bucket(
            self, "DashboardBucket",
            website_index_document="index.html",
            public_read_access=True,
            block_public_access=s3.BlockPublicAccess(
                block_public_acls=False,
                block_public_policy=False,
                ignore_public_acls=False,
                restrict_public_buckets=False,
            ),
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        s3_deployment.BucketDeployment(
            self, "DashboardDeployment",
            sources=[s3_deployment.Source.asset(str(DASHBOARD_DIR))],
            destination_bucket=self.dashboard_bucket,
        )

        # --- Outputs an operator needs to actually run the system ------------
        CfnOutput(self, "ApiUrl", value=self.api.url,
                  description="API Gateway base URL for the capture client and dashboard")
        CfnOutput(self, "ImagesBucketName", value=self.images_bucket.bucket_name,
                  description="S3 bucket the capture client uploads frames to")
        CfnOutput(self, "StateMachineArn", value=self.state_machine.state_machine_arn,
                  description="Inspection state machine ARN")
        CfnOutput(self, "DashboardUrl", value=self.dashboard_bucket.bucket_website_url,
                  description="Static dashboard website URL")
