with import <nixpkgs> {};

let
  tensorflow = pkgs.python37Packages.tensorflow_2.override {
    avx2Support = true;
    fmaSupport = true;
    sse42Support = true;
  };
in stdenv.mkDerivation {
  name = "autokey-py-env";

  buildInputs = with pkgs; [
    ffmpeg-full
    python37Packages.opencv4
    tensorflow
  ];
}

