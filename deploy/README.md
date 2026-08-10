# Deploy

The site is static. `python -m bayline app` writes `app/`, which is committed, and the
root `vercel.json` points a Vercel project at it — connect the repository in the Vercel
dashboard and there is nothing else to configure. No build step, no dependencies.

## The pinned deploy in this directory

This container cannot reach `*.vercel.app` and has no Vercel CLI token, so the first
production deploy went through the Vercel API as a direct file upload. Rather than upload
a 448 KB `index.html` through that channel, the uploaded project is the three files here:
a build script that fetches the committed artefact **at a pinned commit** and refuses to
publish anything whose SHA-256 does not match.

That is worth keeping even once Git integration is set up. The engine's output is
reproducible — two clean runs give byte-identical files — so pinning the hash makes the
deployed page provably the page the engine produced, rather than whatever the branch tip
happened to contain at build time.

`REF` and the three hashes must be updated whenever the artefact is rebuilt:

```bash
git rev-parse HEAD
sha256sum app/index.html app/app.js app/app.css
```

A stale pin fails the build loudly instead of shipping the wrong page, which is the
intended behaviour.
