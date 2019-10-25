import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deep_privacy",
    version="0.0.1",
    author="jugi92",
#   author_email="author@example.com",
    description="Deep Privacy package based on code from: https://github.com/hukkelas/DeepPrivacy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jugi92/DeepPrivacy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
