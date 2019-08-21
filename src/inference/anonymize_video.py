from src.inference import deep_privacy_anonymizer

if __name__ == "__main__":
    generator, imsize, source_path, image_paths, save_path = infer.read_args()
    a = deep_privacy_anonymizer.DeepPrivacyAnonymizer(generator, 128,
                                                      use_static_z=True)
    a.anonymize_video("test_examples/video/FaceOff_Scene1.mp4",
                      "test_examples/video/FaceOff_Scene1_anonymized.mp4", 0, None, True)
