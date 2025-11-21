# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
# BSD 3-Clause Clear License (see LICENSE file)

"""Allow running as: python -m compressai.cli"""

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())
