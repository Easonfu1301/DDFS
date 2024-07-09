from setuptools import setup, find_packages

setup(
    name='DDFS',
    version='1.0.0',
    packages=find_packages(),
    py_modules=['analytic_method',
                'element',
                'kalman_method',
                'cal_dermat',
                'write_root',
                'common_test_parameter',
                'common_test_parameter2',
                'design_detector'],
    install_requires=[
        'numpy<2.0.0',
        'matplotlib==3.8.4',
        'uproot',
        'tqdm',
        'pandas',
        'scipy',
        'filterpy',
    ],
    url='',
    license='',
    author='Eason_fu',
    author_email='1113873713@qq.com',
    description=''
)
# >python setup.py sdist
