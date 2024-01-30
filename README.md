# Robotic Screw Theory Toolkit

## Overview
This repository contains Python code and utilities for advanced robotic manipulations using the principles of screw theory. It is designed to assist in the computational aspects of kinematics and dynamics in robotic systems, providing tools for forward kinematics, Jacobian computation, and inverse kinematics.

## Features
- **Screw Theory Calculations**: Functions to compute screw axes, transformations, and other key elements in screw theory.
- **Forward Kinematics (FK)**: Implementation of the Product of Exponentials (PoE) method.
- **Jacobian Computation**: Tools for calculating both space and body Jacobians.
- **Inverse Kinematics**: Algorithms to solve for joint angles given end-effector positions and orientations.

## Installation
To use this toolkit, clone the repository and install the required dependencies:

```bash
git clone https://github.com/WikiGenius/RoboticScrewTheoryToolkit.git
cd RoboticScrewTheoryToolkit
# Assuming you're using Poetry for dependency management
poetry install
