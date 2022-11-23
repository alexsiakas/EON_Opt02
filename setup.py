
import os

from setuptools import setup


package = 'eon_opt02'

os.chdir(os.path.abspath(os.path.dirname(__file__)))
version = open(os.path.join(package, '__version__.txt')).read()
install_requires = open('requirements.txt').read().split('\n')



setup(
    name=package,
    version=version,
    # license='MIT',
    # classifiers=['Development Status :: 4 - Beta',
    #              'Environment :: Console',
    #              'Intended Audience :: Science/Research',
    #              'Topic :: Scientific/Engineering :: Astronomy',
    #              'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    #              'Operating System :: MacOS :: MacOS X',
    #              'Programming Language :: Python :: 3.8',
    #              ],
    entry_points={},
    packages=[package],
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)
