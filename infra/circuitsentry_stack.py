from aws_cdk import Stack, RemovalPolicy, Duration
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
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
