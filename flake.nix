{
  description = "Catgrad - A Categorical Deep Learning Compiler";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }: let
    defaultCudaCapability = "89";
    defaultCudaPackages = pkgs: pkgs.cudaPackages_13;

    mkCatgrad = pkgs: {
      withExamples ? true,
      withMlir ? false,
      llvmPackages ? pkgs.llvmPackages_21,
      withCuda ? false,
      cudaPackages ? defaultCudaPackages pkgs,
      cudaCapability ? defaultCudaCapability,
    }: let
      manifest = (pkgs.lib.importTOML ./Cargo.toml).workspace.package;
      mlirInputs = with llvmPackages; [mlir llvm clang];
      cudaInputs = with cudaPackages; [
        cuda_cccl
        cuda_cudart
        cuda_nvrtc
        libcublas
        libcurand
      ];
    in
      pkgs.rustPlatform.buildRustPackage {
        pname = "catgrad";
        version = manifest.version;
        src = self;
        cargoDeps = pkgs.rustPlatform.importCargoLock {
          lockFile = ./Cargo.lock;
        };

        auditable = false;

        cargoBuildFlags =
          ["--workspace"]
          ++ pkgs.lib.optionals withExamples ["--examples"]
          ++ pkgs.lib.optionals withCuda ["--features" "catgrad/cuda"];

        nativeBuildInputs =
          pkgs.lib.optionals (withMlir || withCuda) [pkgs.makeWrapper]
          ++ pkgs.lib.optionals withCuda [cudaPackages.cuda_nvcc];

        buildInputs =
          pkgs.lib.optionals withMlir mlirInputs
          ++ pkgs.lib.optionals withCuda cudaInputs;

        CUDA_COMPUTE_CAP = pkgs.lib.optionalString withCuda cudaCapability;
        CUDA_TOOLKIT_ROOT_DIR = pkgs.lib.optionalString withCuda (pkgs.lib.getDev cudaPackages.cuda_cudart);

        doCheck = !withCuda;

        postInstall =
          pkgs.lib.optionalString withExamples ''
            mkdir -p $out/bin
            find target -path '*/release/examples/*' -executable -type f \
              ! -name '*-????????????????' \
              -exec install -Dm755 {} $out/bin/ \;
          ''
          + pkgs.lib.optionalString withMlir ''
            if [ -x "$out/bin/mlir-llm" ]; then
              wrapProgram "$out/bin/mlir-llm" \
                --prefix PATH : "${pkgs.lib.makeBinPath mlirInputs}" \
                --prefix LIBRARY_PATH : "${pkgs.lib.makeLibraryPath mlirInputs}" \
                --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath mlirInputs}" \
                --prefix DYLD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath mlirInputs}" \
                --prefix NIX_LDFLAGS " " "-L${pkgs.lib.makeLibraryPath mlirInputs}"
            fi
          ''
          + pkgs.lib.optionalString withCuda ''
            for bin in $out/bin/*; do
              if [ -x "$bin" ] && [ ! -L "$bin" ]; then
                wrapProgram "$bin" \
                  --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath cudaInputs}"
              fi
            done
          '';

        meta = with pkgs.lib; {
          description = manifest.description;
          license = licenses.mit;
          mainProgram =
            if withMlir
            then "mlir-llm"
            else "llama";
        };
      };
  in
    {
      overlays.default = final: _prev: {
        catgrad = mkCatgrad final {
          withCuda = final.config.cudaSupport or false;
        };
      };
    }
    // flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in {
      packages = {
        default = mkCatgrad pkgs {};
        minimal = mkCatgrad pkgs {withExamples = false;};
        withMlir = mkCatgrad pkgs {withMlir = true;};
        withCuda = mkCatgrad pkgs {withCuda = true;};
      };

      devShells.cuda = pkgs.mkShell {
        inputsFrom = [(mkCatgrad pkgs {withCuda = true;})];
        CUDA_COMPUTE_CAP = defaultCudaCapability;
        CUDA_TOOLKIT_ROOT_DIR = pkgs.lib.getDev (defaultCudaPackages pkgs).cuda_cudart;
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          "/run/opengl-driver"
        ];
      };
    });
}
