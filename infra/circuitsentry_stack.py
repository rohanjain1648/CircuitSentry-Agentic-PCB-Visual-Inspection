from aws_cdk import Stack, RemovalPolicy, Duration
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from aws_cdk import aws_apigateway as apigw
from constructs import Construct


class CircuitSentryStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

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
            code=_lambda.Code.from_asset("circuitsentry"),
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

        pass_state = sfn.Pass(self, "Pass")
        flag_state = sfn.Pass(self, "FlagForApproval")
        recapture_state = sfn.Pass(self, "Recapture")
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
            code=_lambda.Code.from_asset("circuitsentry"),
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
            code=_lambda.Code.from_asset("circuitsentry"),
            environment={"RUNS_TABLE_NAME": self.runs_table.table_name},
        )
        runs_detail_fn = _lambda.Function(
            self, "RunsDetailFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="circuitsentry.backend.lambda_api.runs.detail_handler",
            code=_lambda.Code.from_asset("circuitsentry"),
            environment={"RUNS_TABLE_NAME": self.runs_table.table_name},
        )
        approve_fn = _lambda.Function(
            self, "ApproveFn",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="circuitsentry.backend.lambda_api.approve.lambda_handler",
            code=_lambda.Code.from_asset("circuitsentry"),
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
