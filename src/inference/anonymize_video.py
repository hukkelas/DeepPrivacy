from src.inference import deep_privacy_anonymizer, infer

if __name__ == "__main__":
    generator, imsize, source_path, image_paths, save_path = infer.read_args()
    a = deep_privacy_anonymizer.DeepPrivacyAnonymizer(generator, 32,
                                                      use_static_z=True,
                                                      keypoint_threshold=.1,
                                                      face_threshold=.3)
    a.anonymize_video("test_examples/video/selfie2.mp4",
                      "test_examples/video/selfie2_anonymized.mp4", 0, None, True)
