def main(_):
    model = ViTClassifier()
    trainer = pl.Trainer(
        default_root_dit = "",
        gpus = 1,
        max_epochs = FLAGS.epochs,
        precision = 16 
    )

if __name__ == "__main__":
    main()

