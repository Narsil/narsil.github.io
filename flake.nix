{
  description = "Dev shell for Jekyll blog with Ruby, Bundler, and gems";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pkgs.ruby_3_1
          pkgs.bundler
          pkgs.git
          pkgs.libffi
          pkgs.zlib
          pkgs.pkg-config
          pkgs.gcc
          pkgs.makeWrapper
          pkgs.nodejs_20
          pkgs.python3Packages.python
          pkgs.python3Packages.graphviz
          pkgs.graphviz
          pkgs.fg-virgil
        ];

        shellHook = ''
          export GEM_HOME=$PWD/.gems
          export PATH=$GEM_HOME/bin:$PATH

          export DOTFONTPATH=${pkgs.fg-virgil}/share/fonts/

          if ! command -v bundle > /dev/null; then
            gem install bundler
          fi

          if [ ! -d ".gems" ] || [ Gemfile.lock -nt .gems ]; then
            bundle install --path=.gems
          fi

          echo "Ruby, Bundler, Node.js, Python, Graphviz, Virgil font, and all gems are ready!"
        '';
      };
    };
}
