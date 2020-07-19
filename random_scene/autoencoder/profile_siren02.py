from __future__ import print_function
from siren02 import *
import cProfile
import pstats

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(threadName)s] %(message)s')
    args_input = ["--log-file", "profile_siren02.log", "--dataset", "screens4_256P",
                  "--dataset-seed", "335248", "--img-batch-size", "64", "--batch-size", "16384",
                  "--random-position", "--random-steps", "4", "--random-max-t", "1.0"]
    args = arg_parser(args_input, "02")

    if len(args.log_file) > 0:
        log_formatter = logging.Formatter("%(asctime)s: %(message)s")
        root_logger = logging.getLogger()

        file_handler = logging.FileHandler(join(args.model_path, args.log_file))
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_loader, train_loader, = get_data_loaders(args, device)
    model = Siren(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
                  pos_encoding_levels=(args.pos_encoding, args.rot_encoding), dropout=args.dropout)
    if len(args.load_model_state) > 0:
        model_path = os.path.join(args.model_path, args.load_model_state)
        if os.path.exists(model_path):
            logging.info("Loading model from {}".format(model_path))
            model.load_state_dict(torch.load(model_path))
            model.eval()
    model = model.to(device, dtype=torch.float32)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.schedule_step_size, gamma=args.schedule_gamma)
    criterion = ModelLoss(device=device)

    stats = join(args.model_path, "pstats")
    cProfile.run("train(args, model, device, train_loader, criterion, optimizer, 0)", stats)
    logging.info("Profiling complete!")
    p = pstats.Stats(stats)
    p.strip_dirs().sort_stats(1).print_stats()
    logging.info("Done!")
