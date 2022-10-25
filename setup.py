import setuptools

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name='lava-lol',
    version='3.0.0',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'opencv-python>=4.6.0.66',
        'Flask==2.0.0',
        'Jinja2==3.0.0',
        'kaleido==0.2.1',
        'numpy>=1.21.6',
        'pandas==1.1.5',
        'Pillow==9.0.1',
        'plotly==4.14.3',
        'pytube==10.8.1',
        'scipy>=1.5.4',
        'sympy==1.8',
        'yt-dlp==2022.10.4'
    ],
    download_url='https://github.com/Steeeephen/LAVA/archive/refs/tags/3.0.0.tar.gz'
)