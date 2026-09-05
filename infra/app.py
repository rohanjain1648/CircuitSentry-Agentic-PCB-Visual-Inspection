import aws_cdk as cdk
from circuitsentry_stack import CircuitSentryStack

app = cdk.App()
CircuitSentryStack(app, "CircuitSentryStack")
app.synth()
