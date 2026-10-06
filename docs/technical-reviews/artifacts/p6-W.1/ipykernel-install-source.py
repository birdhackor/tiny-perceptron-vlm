class InstallIPythonKernelSpecApp(Application):
    """Dummy app wrapping argparse"""

    name = Unicode("ipython-kernel-install")

    def initialize(self, argv: list[str] | None = None) -> None:
        """Initialize the app."""
        if argv is None:
            argv = sys.argv[1:]
        self.argv = argv

    def start(self) -> None:
        """Start the app."""
        import argparse

        parser = argparse.ArgumentParser(
            prog=self.name, description="Install the IPython kernel spec."
        )
        parser.add_argument(
            "--user",
            action="store_true",
            help="Install for the current user instead of system-wide",
        )
        parser.add_argument(
            "--name",
            type=str,
            default=KERNEL_NAME,
            help="Specify a name for the kernelspec."
            " This is needed to have multiple IPython kernels at the same time.",
        )
        parser.add_argument(
            "--display-name",
            type=str,
            help="Specify the display name for the kernelspec."
            " This is helpful when you have multiple IPython kernels.",
        )
        parser.add_argument(
            "--profile",
            type=str,
            help="Specify an IPython profile to load. "
            "This can be used to create custom versions of the kernel.",
        )
        parser.add_argument(
            "--prefix",
            type=str,
            help="Specify an install prefix for the kernelspec."
            " This is needed to install into a non-default location, such as a conda/virtual-env.",
        )
        parser.add_argument(
            "--sys-prefix",
            action="store_const",
            const=sys.prefix,
            dest="prefix",
            help="Install to Python's sys.prefix."
            " Shorthand for --prefix='%s'. For use in conda/virtual-envs." % sys.prefix,
        )
        parser.add_argument(
            "--env",
            action="append",
            nargs=2,
            metavar=("ENV", "VALUE"),
            help="Set environment variables for the kernel.",
        )
        parser.add_argument(
            "--frozen_modules",
            action="store_true",
            help="Enable frozen modules for potentially faster startup."
            " This has a downside of preventing the debugger from navigating to certain built-in modules.",
        )
        opts = parser.parse_args(self.argv)
        if opts.env:
            opts.env = dict(opts.env)
        try:
            dest = install(
                user=opts.user,
                kernel_name=opts.name,
                profile=opts.profile,
                prefix=opts.prefix,
                display_name=opts.display_name,
                env=opts.env,
            )
        except OSError as e:
            if e.errno == errno.EACCES:
                print(e, file=sys.stderr)
                if opts.user:
                    print("Perhaps you want `sudo` or `--user`?", file=sys.stderr)
                self.exit(1)
            raise
        print(f"Installed kernelspec {opts.name} in {dest}")

def install(
    kernel_spec_manager: KernelSpecManager | None = None,
    user: bool = False,
    kernel_name: str = KERNEL_NAME,
    display_name: str | None = None,
    prefix: str | None = None,
    profile: str | None = None,
    env: dict[str, str] | None = None,
    frozen_modules: bool = False,
) -> str:
    """Install the IPython kernelspec for Jupyter

    Parameters
    ----------
    kernel_spec_manager : KernelSpecManager [optional]
        A KernelSpecManager to use for installation.
        If none provided, a default instance will be created.
    user : bool [default: False]
        Whether to do a user-only install, or system-wide.
    kernel_name : str, optional
        Specify a name for the kernelspec.
        This is needed for having multiple IPython kernels for different environments.
    display_name : str, optional
        Specify the display name for the kernelspec
    profile : str, optional
        Specify a custom profile to be loaded by the kernel.
    prefix : str, optional
        Specify an install prefix for the kernelspec.
        This is needed to install into a non-default location, such as a conda/virtual-env.
    env : dict, optional
        A dictionary of extra environment variables for the kernel.
        These will be added to the current environment variables before the
        kernel is started
    frozen_modules : bool, optional
        Whether to use frozen modules for potentially faster kernel startup.
        Using frozen modules prevents debugging inside of some built-in
        Python modules, such as io, abc, posixpath, ntpath, or stat.
        The frozen modules are used in CPython for faster interpreter startup.
        Ignored for cPython <3.11 and for other Python implementations.

    Returns
    -------
    The path where the kernelspec was installed.
    """
    if kernel_spec_manager is None:
        kernel_spec_manager = KernelSpecManager()

    if env is None:
        env = {}

    if (kernel_name != KERNEL_NAME) and (display_name is None):
        # kernel_name is specified and display_name is not
        # default display_name to kernel_name
        display_name = kernel_name
    overrides: dict[str, Any] = {}
    if display_name:
        overrides["display_name"] = display_name
    if profile:
        extra_arguments = ["--profile", profile]
        if not display_name:
            # add the profile to the default display name
            overrides["display_name"] = "Python %i [profile=%s]" % (sys.version_info[0], profile)
    else:
        extra_arguments = None

    python_arguments = None

    # addresses the debugger warning from debugpy about frozen modules
    if platform.python_implementation() == "CPython":
        if not frozen_modules:
            # disable frozen modules
            python_arguments = ["-Xfrozen_modules=off"]
        elif "PYDEVD_DISABLE_FILE_VALIDATION" not in env:
            # user opted-in to have frozen modules, and we warned them about
            # consequences for the - disable the debugger warning
            env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

    if env:
        overrides["env"] = env
    path = write_kernel_spec(
        overrides=overrides, extra_arguments=extra_arguments, python_arguments=python_arguments
    )
    dest = kernel_spec_manager.install_kernel_spec(
        path, kernel_name=kernel_name, user=user, prefix=prefix
    )
    # cleanup afterward
    shutil.rmtree(path)
    return dest

def get_kernel_dict(
    extra_arguments: list[str] | None = None, python_arguments: list[str] | None = None
) -> dict[str, Any]:
    """Construct dict for kernel.json"""
    return {
        "argv": make_ipkernel_cmd(
            extra_arguments=extra_arguments, python_arguments=python_arguments
        ),
        "display_name": "Python %i (ipykernel)" % sys.version_info[0],
        "language": "python",
        "metadata": {"debugger": True, "supported_encryption": ["curve"]},
        "kernel_protocol_version": "5.5",
    }
