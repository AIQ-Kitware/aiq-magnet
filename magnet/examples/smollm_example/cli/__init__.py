"""
The executables this example's nodes run.

Separated from the scaffolding on purpose. Everything in here is an ordinary
command-line program: it reads files, talks to an endpoint, writes JSON, and
knows nothing about kwdagger, containers or leases. ``pipeline.py`` beside this
package is what turns them into a DAG.

That split is the point of the example as much as the leasing is. A node's
executable should be runnable by hand, and debuggable without a scheduler.
"""
