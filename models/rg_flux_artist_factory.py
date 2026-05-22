from models.flux_sr_artist import FluxSRArtist


def get_flux_backend(config):
    model_config = config.get("model", {}) if isinstance(config, dict) else {}
    backend = model_config.get("flux_backend", "flux1")
    return str(backend or "flux1").lower()


def build_rg_flux_artist(config):
    backend = get_flux_backend(config)
    if backend in {"flux1", "flux_1", "flx1"}:
        return FluxSRArtist(config)
    if backend in {"flux2_klein", "flux2-klein", "flux_2_klein"}:
        from models.flux2_klein_sr_artist import Flux2KleinSRArtist

        return Flux2KleinSRArtist(config)
    raise ValueError(f"Unsupported RG-FLUX backend: {backend}")
