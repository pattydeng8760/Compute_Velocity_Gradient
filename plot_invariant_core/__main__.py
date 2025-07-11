from .plot_invariant import parse_arguments, PlotInvariant
from .utils import init_logging_from_cut

def main(args=None):
    # 1) Parse CLI args
    if args is None:
        args = parse_arguments()

    # 2) Redirect stdout to log file and set up logging
    init_logging_from_cut(args.cut, args.data_type)

    # 3) Build and run
    runner = PlotInvariant(args)
    runner.run()

if __name__ == "__main__":
    main()