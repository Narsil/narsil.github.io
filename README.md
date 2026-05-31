# narsil.github.io

Source for [nodata.dev](https://nodata.dev) — a small blog about ML and software development.

## Build and serve locally

```
nix develop -c zola serve
```

Then open <http://127.0.0.1:1111>.

## Layout

```
config.toml            Zola configuration
content/               Posts (one .md per article) + about/, search/ sections
templates/             Site-level template overrides (extend the pickles theme)
sass/custom.scss       Custom styles, including dark-mode overrides
static/                Files served verbatim at the site root (assets/, images/, CNAME, 404.html)
themes/zola-pickles    Theme submodule
scripts/               Build helpers (flatten-html.sh, generate_llm_bottlenecks_diagram.py)
.github/workflows/     CI: builds with Zola, flattens slug.html dirs, deploys to gh-pages
```

## Deployment

Pushing to `master` triggers `.github/workflows/deploy-zola.yml`, which builds the
site, flattens `slug.html/index.html` files to flat `slug.html` (matching the legacy
Jekyll permalink layout), and publishes `public/` to the `gh-pages` branch served at
`nodata.dev`.
