"""PyInstaller 打包入口。

打包后由 Electron 主进程以 ``extraResources/physio_service/physio_service`` 启动，
目标机器无需安装 Python。
"""
from physio_service.server import main

if __name__ == "__main__":
    main()
