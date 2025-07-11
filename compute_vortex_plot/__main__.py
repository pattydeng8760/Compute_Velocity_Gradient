from .vortex_plot import parse_arguments, VortexPlot
from .utils import init_logging_from_cut

def main(args=None):
    """Main entry point for the vortex plot module."""
    # Parse CLI arguments
    if args is None:
        args = parse_arguments()

    # Initialize logging
    init_logging_from_cut(args.cut, args.data_type)

    # Create and run VortexPlot instance
    runner = VortexPlot(args)
    runner.run()

if __name__ == "__main__":
    main()