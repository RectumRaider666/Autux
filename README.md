# Autux

Auto-Termux Custom Name-Space

## Description

Autux is a tool for customizing your Termux namespace.

## Installation

### Via Termux Package Manager

To install Autux via the Termux package manager, you need to build and install the package from the termux-packages repository:

1. Clone the termux-packages repository:
   ```bash
   git clone https://github.com/termux/termux-packages.git
   cd termux-packages
   ```

2. Copy the autux package folder from this repository to the `packages/` directory of termux-packages.

3. Build and install the package:
   ```bash
   ./build-package.sh -I autux
   ```

### Manual Installation

Alternatively, you can install Autux manually:

```bash
git clone https://github.com/RectumRaider666/Autux.git
cd Autux
cp packages/autux/autux $PREFIX/bin/
chmod +x $PREFIX/bin/autux
```

## Usage

```bash
autux --help
autux --version
```

## License

MIT
