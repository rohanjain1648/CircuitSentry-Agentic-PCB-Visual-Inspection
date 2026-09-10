"""CDK entrypoint.

Run via `cdk synth` / `cdk deploy` from the repository root; cdk.json invokes
this as `python -m infra.app`, which puts the repo root on sys.path so the
package-style import below resolves (the same way the tests import it) and so
the Lambda asset root in circuitsentry_stack.py stays correct.
"""
import aws_cdk as cdk

from infra.circuitsentry_stack import CircuitSentryStack

app = cdk.App()
# bundle_dependencies=True: real deploys need opencv-python-headless/numpy
# installed into the Lambda package via Docker (see circuitsentry_stack.py).
# The test suite instantiates CircuitSentryStack directly and leaves this at
# its default (False) so tests don't require Docker.
CircuitSentryStack(app, "CircuitSentryStack", bundle_dependencies=True)
app.synth()
