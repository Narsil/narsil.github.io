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
scripts/               Content helpers (e.g. generate_llm_bottlenecks_diagram.py)
.github/workflows/     CI: builds with Zola, deploys via actions/deploy-pages
```

## Deployment

Pushing to `master` triggers `.github/workflows/deploy-zola.yml`: install Zola,
`zola build`, upload `public/` as a Pages artifact, and deploy via
`actions/deploy-pages`. Old Jekyll permalinks (`/YYYY/MM/DD/slug.html`) keep
resolving because each post declares them as `aliases` — Zola emits a meta-refresh
redirect HTML at each alias path.
