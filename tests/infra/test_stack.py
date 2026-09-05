import aws_cdk as cdk
from aws_cdk.assertions import Template
from infra.circuitsentry_stack import CircuitSentryStack


def test_stack_has_s3_bucket_and_dynamodb_table():
    app = cdk.App()
    stack = CircuitSentryStack(app, "TestStack")
    template = Template.from_stack(stack)

    template.resource_count_is("AWS::S3::Bucket", 1)
    template.has_resource_properties("AWS::DynamoDB::Table", {
        "KeySchema": [
            {"AttributeName": "run_id", "KeyType": "HASH"},
            {"AttributeName": "seq", "KeyType": "RANGE"},
        ],
    })
