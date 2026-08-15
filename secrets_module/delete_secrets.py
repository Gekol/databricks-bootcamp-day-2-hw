# Using Python with databricks-sdk
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
w.secrets.delete_scope(scope="hw-db")