import faulthandler
import sys

faulthandler.dump_traceback_later(90, exit=True)
from context_eval.run_eval import main

raise SystemExit(main(sys.argv[1:]))
