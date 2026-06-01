{
  description = "narsil.github.io — Zola source, build, and dev shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  # Theme is wired in as a flake input rather than via the git submodule so
  # `nix build` is reproducible from a clean checkout (the submodule path is a
  # gitlink Nix's source filter doesn't follow).
  inputs.zola-pickles = {
    url = "github:lukehsiao/zola-pickles";
    flake = false;
  };

  outputs =
    { self, nixpkgs, zola-pickles }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems =
        f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));

      mkApp = pkg: {
        type = "app";
        program = "${pkg}/bin/${pkg.meta.mainProgram or pkg.pname}";
      };
    in
    {
      packages = forAllSystems (pkgs: rec {
        default = site;

        site = pkgs.stdenv.mkDerivation {
          pname = "narsil-blog";
          version = self.shortRev or self.dirtyShortRev or "dirty";
          src = nixpkgs.lib.cleanSource ./.;
          nativeBuildInputs = [ pkgs.zola ];
          buildPhase = ''
            runHook preBuild
            # The theme submodule isn't part of the Nix source closure, but
            # an empty submodule-placeholder directory IS — remove it before
            # copying the flake input in, otherwise `cp -r` nests the theme
            # inside the placeholder (themes/zola-pickles/zola-pickles/…).
            rm -rf themes/zola-pickles
            mkdir -p themes
            cp -r --no-preserve=mode ${zola-pickles} themes/zola-pickles
            zola build --output-dir $out
            runHook postBuild
          '';
          dontInstall = true;
        };
      });

      apps = forAllSystems (pkgs: rec {
        default = serve;

        # `nix run .` (or `nix run .#serve`) — dev server with live reload.
        # Forwards any extra args (e.g. `nix run .#serve -- --port 1117`).
        serve = mkApp (pkgs.writeShellApplication {
          name = "serve";
          runtimeInputs = [ pkgs.zola ];
          text = ''
            exec zola serve "$@"
          '';
        });

        # `nix run .#check` — link/anchor checker.
        check = mkApp (pkgs.writeShellApplication {
          name = "check";
          runtimeInputs = [ pkgs.zola ];
          text = ''
            exec zola check "$@"
          '';
        });
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          buildInputs = [
            pkgs.zola
            pkgs.git
          ];

          shellHook = ''
            echo "Zola $(zola --version | awk '{print $2}') ready. Run: zola serve"
          '';
        };
      });
    };
}
