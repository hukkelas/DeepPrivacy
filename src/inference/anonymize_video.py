from src.inference import deep_privacy_anonymizer, infer

if __name__ == "__main__":
    generator, _, source_path, _, target_path, config = infer.read_args(
        [{"name": "anonymize_source", "default": False},
        {"name": "max_face_size", "default": 1.0}],
    )
    a = deep_privacy_anonymizer.DeepPrivacyAnonymizer(generator, 32,
                                                      use_static_z=True,
                                                      keypoint_threshold=.3,
                                                      face_threshold=.9)

    a.anonymize_video(source_path,
                      target_path, 
                      start_frame=0,
                      end_frame=None,
                      with_keypoints=True,
                      anonymize_source=config.anonymize_source,
                      max_face_size=float(config.max_face_size))
