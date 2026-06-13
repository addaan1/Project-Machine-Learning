# Indonesia Economic Map Redesign

## Goal

Redesign the EcoDash Indonesia economic map into a light-premium hybrid visualization that uses the existing 38-province GeoJSON as the geographic source of truth.

## Visual Direction

- Keep the existing EcoDash light interface and blue accent system.
- Use true province boundaries from `indonesia_provinces.geojson`.
- Make choropleth color the primary encoding.
- Add restrained heat glow only around high-value province centroids.
- Keep the basemap quiet so province colors remain legible.
- Use crisp borders, subtle depth, and restrained motion rather than decorative clutter.

## Workspace

The map workspace will contain:

- A compact floating toolbar for metric selection, data year, province count, and reset view.
- The full Indonesia map as the dominant surface.
- A floating national-summary panel with average, highest, and lowest values.
- A top-five ranking panel whose rows focus the matching province when selected.
- A continuous legend based on quantile thresholds.
- A responsive lower sheet on small screens so controls do not obscure the map.

## Map Layers

1. A neutral CARTO light basemap.
2. A Leaflet GeoJSON choropleth layer for all 38 provinces.
3. A low-opacity heat layer generated from GeoJSON feature centroids for upper-quantile values.
4. Interactive province outlines and popup content.

The choropleth remains readable if the heat layer is disabled or unsupported.

## Data Behavior

- Continue using `/api/metrics-latest/`; no API contract changes are required.
- Match GeoJSON `properties.name` directly to API province names.
- Calculate min, max, mean, ranks, and quantile thresholds per selected metric.
- Use sequential blue tones for most values, cyan for upper-middle values, amber for high values, and coral for the highest quantile.
- Display the source year returned by the API.

## Interaction

- Hover increases border contrast and shows a compact tooltip.
- Click opens a popup with province value, national comparison, and rank.
- Ranking rows focus the selected province and open its popup.
- Metric changes update fills, glow, legend, rankings, and summary with a short fade transition.
- Reset restores the Indonesia-wide viewport.

## Error And Empty States

- Show a calm in-map error panel when API or GeoJSON loading fails.
- Render provinces with a neutral fill if a metric value is missing.
- Keep the map usable when the heat plugin is unavailable.

## Responsive Behavior

- Desktop: toolbar at top-left, summary and ranking on the right, legend at bottom-left.
- Tablet: narrower right panel and reduced map chrome.
- Mobile: stacked toolbar above the map, compact legend, and ranking as a scrollable panel below the map.

## Verification

- Django template test confirms the page references the local 38-province GeoJSON and required hybrid-map UI elements.
- Browser verification confirms all 38 features render, metric switching works, popup/ranking interactions work, and no console errors occur.
- Verify desktop and mobile viewport layouts.

## Scope

Only the map presentation and its page-level tests are changed. The data API, datasets, navigation, and unrelated pages remain untouched.
