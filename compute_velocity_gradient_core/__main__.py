from .velocity_invariant import parse_arguments, VelocityInvariant
from .utils import init_logging_from_cut

def main(args=None):
    # 1) Parse CLI args
    if args is None:
        args = parse_arguments()

    # 2) Redirect stdout to log_<cut>.txt and set up logging
    init_logging_from_cut(args.cut)

    # 3) Build and run
    runner = VelocityInvariant(args)
    runner.run()

if __name__ == "__main__":
    main()