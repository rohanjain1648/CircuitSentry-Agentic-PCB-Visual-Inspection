import aws_cdk as cdk
from aws_cdk.assertions import Template
from infra.circuitsentry_stack import CircuitSentryStack


def test_stack_has_s3_bucket_and_dynamodb_table():
    app = cdk.App()
    stack = CircuitSentryStack(app, "TestStack")
    template = Template.from_stack(stack)

    # Images bucket + dashboard static-site bucket.
    template.resource_count_is("AWS::S3::Bucket", 2)
    template.has_resource_properties("AWS::DynamoDB::Table", {
        "KeySchema": [
            {"AttributeName": "run_id", "KeyType": "HASH"},
            {"AttributeName": "seq", "KeyType": "RANGE"},
        ],
    })


def test_stack_has_dashboard_site_bucket_and_operator_outputs():
    app = cdk.App()
    stack = CircuitSentryStack(app, "TestStack")
    template = Template.from_stack(stack)

    # The dashboard bucket is configured as a static website (spec §5).
    template.has_resource_properties("AWS::S3::Bucket", {
        "WebsiteConfiguration": {"IndexDocument": "index.html"},
    })
    # ...and its contents are deployed by a BucketDeployment custom resource.
    template.resource_count_is("Custom::CDKBucketDeployment", 1)

    outputs = template.find_outputs("*")
    assert {"ApiUrl", "ImagesBucketName", "StateMachineArn", "DashboardUrl"} <= set(outputs)
