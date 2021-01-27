import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepflow_by_zuo", # Replace with your own username
    version="0.0.1",
    author="zuo",
    author_email="zzuuoo666@sina.com",
    description="This is a mini but function complicated neural networks framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Coding-Zuo/MyDeep-learning-framework",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)