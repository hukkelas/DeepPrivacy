from src.inference import deep_privacy_anonymizer, infer


if __name__ == "__main__":
    generator, imsize, source_path, image_paths, save_path = infer.read_args()

    anonymizer = deep_privacy_anonymizer.DeepPrivacyAnonymizer(
        generator, 128, use_static_z=True)

    anonymizer.anonymize_folder(source_path, save_path)
