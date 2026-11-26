"""ament_python package: launch files, parameters, rviz config and the vo_node entry point."""

from glob import glob

from setuptools import setup

PACKAGE = "metricvision_ros"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=[PACKAGE],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
        (f"share/{PACKAGE}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE}/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="MetricVision",
    maintainer_email="noreply@example.com",
    description="ROS 2 launch and runtime wrapper for MetricVision stereo VO",
    license="MIT",
    entry_points={"console_scripts": [f"vo_node = {PACKAGE}.launcher:main", f"vo_plot = {PACKAGE}.launcher:plot_main"]},
)
