"""CDK entrypoint.

Run via `cdk synth` / `cdk deploy` from the repository root; cdk.json invokes
this as `python -m infra.app`, which puts the repo root on sys.path so the
package-style import below resolves (the same way the tests import it) and so
the Lambda asset root in circuitsentry_stack.py stays correct.
"""
import aws_cdk as cdk

from infra.circuitsentry_stack import CircuitSentryStack

app = cdk.App()
CircuitSentryStack(app, "CircuitSentryStack")
app.synth()
