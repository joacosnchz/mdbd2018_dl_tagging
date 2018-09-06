import time
from datetime import datetime
StartTime= datetime.now()
time.sleep(10)
diff = datetime.now() - StartTime

print(str(diff))