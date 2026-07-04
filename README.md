<div align="center">
    <a href="https://github.com/ergetie/darkstar">
        <img width="150" height="150" src="darkstar/logo.png">
    </a>
    <br>
    <br>
   <div style="display: flex;">
        <a href="https://github.com/ergetie/darkstar/actions/workflows/ci.yml">
            <img src="https://github.com/ergetie/darkstar/actions/workflows/ci.yml/badge.svg">
        </a>
        <a href="https://github.com/ergetie/darkstar/actions/workflows/build-addon.yml">
            <img src="https://github.com/ergetie/darkstar/actions/workflows/build-addon.yml/badge.svg">
        </a>
        <a href="https://github.com/ergetie/darkstar/releases">
            <img src="https://img.shields.io/github/v/release/ergetie/darkstar?label=beta&color=cyan">
        </a>
    </div>
    <h3>Local AI-powered home battery optimization for solar homes</h3>
</div>

> [!WARNING]
> **PUBLIC BETA**: Darkstar is currently in Public Beta. This software controls high-power electrical equipment. Always maintain human supervision and ensure your Home Assistant safety cut-offs are configured. Use at your own risk!

> [!CAUTION]
> If you're using a Raspberry Pi, ensure you have at least a **Raspberry Pi 4** (RPi 5 recommended).

Darkstar is a local, privacy-first energy management system that optimizes your home battery, solar production, and electricity costs using machine learning (LightGBM) and mathematical optimization (MILP).

![Dashboard Preview](docs/images/dashboard-preview.png)

## Features

- **Smart Optimization** — Minimizes electricity costs over a 48-hour rolling horizon
- **ML Forecasting** — Learns your home's load and PV production patterns
- **Real-time Execution** — Automatic inverter control via Home Assistant
- **Beautiful Dashboard** — React-based UI with live schedule visualization
- **Self-Learning** — Parameters auto-tune to your home over time. **New:** Automatic ML model retraining.
- **Smart EV Charging** — Intelligent charging optimization that respects your house battery and prioritizes cheap grid hours.
- **Load Disaggregation** — Separates base load from controllable appliances (Water Heater, EV, etc.) for better ML accuracy
- **Vacation Mode** — Safe anti-legionella water heating while away

## Fork additions (Aksinder)

This fork tracks upstream and adds device-level controllers and pricing features, all
**default-OFF or observe-only**. Highlights:

- **Deferrable smart appliances** — turnkey dishwasher/washer scheduling from a power
  sensor + smart plug (auto-arm, cheapest *forecast* window, done detection; observe-first
  with opt-in plug gating — pause at start only, running cycles never interrupted)
- **Real-time EV surplus charging** — per-tick current control (solar surplus → cheap
  grid → conditional battery assist), target-SoC caps, departure deadlines, vacation targets
- **EV SoC estimator** — self-calibrating dead-reckoning for cars with no API
- **Thermal water tanks** — hot-water litres/level estimation from the heating-power
  meter alone (no temperature probe), with per-tank vacation partitioning
- **Single-source pricing** — import fees and export compensation (påslag, nätnytta)
  driven by HA helpers, öre/SEK auto-conversion, VAT-inclusive fee support
  (`fees_include_vat`), negative-price export curtailment
- **Multi-inverter AC-limit modeling** — `system.solar_arrays[].on_battery_inverter` tags so a
  hybrid + PV-only inverter site plans battery discharge correctly
- **Load priority (WTP) tiers** — reservation prices per load, incl. dynamic
  percentile caps; **phase-aware load modeling**; excess-PV climate sink; unknown-load
  observability

Full feature docs: [`darkstar-dev/DOCS.md`](darkstar-dev/DOCS.md) · design notes: [`docs/designs/`](docs/designs/)


## Installation

### Option 1: Home Assistant Add-on (Recommended)

1. Go to **[Settings → Add-ons → Add-on store](https://my.home-assistant.io/redirect/supervisor_store/)**, click **⋮ → Repositories**, fill in</br> `https://github.com/ergetie/darkstar` and click **Add → Close** or click the **Add repository** button below.
   [![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fergetie%2Fdarkstar)
2. Click on **Darkstar Energy Manager** and press **Install**.
3. Start the add-on - Darkstar will **automatically detect** your Home Assistant connection. No manual token required!
4. Click **OPEN WEB UI** and navigate to **Settings** in the sidebar to map your sensors.

### [DEV] Development Version (Optional)

For testers and developers who want the latest features (and bugs!) from the `dev` branch:
1. Follow the installation steps above to add the repository.
2. In the Add-on Store, look for **[DEV] Darkstar Energy Manager**.
3. **Note**: The Dev version is optimized for **amd64** architecture and updates frequently. It can be installed alongside the stable version for testing.

### Option 2: Docker Compose

1. Copy `config.default.yaml` to `config.yaml` and `secrets.example.yaml` to `secrets.yaml`.
2. Edit `secrets.yaml` with your Home Assistant credentials.
3. Start with `docker-compose up -d`.
4. Access the UI at **http://localhost:5000**.

---

## Getting Started

Once you have installed Darkstar, follow the comprehensive guide to tune the system for your home:

[**👉 Read the Configuration Guide**](docs/SETUP_GUIDE.md)

Already configured? Learn how to operate the system daily:

[**📘 Read the User Manual**](docs/USER_MANUAL.md)

---

## Configuration

### `config.yaml`

Main configuration for your system:

```yaml
# Solar arrays (Up to 6 supported)
system:
  solar_arrays:
    - name: "Roof South"
      azimuth: 180
      tilt: 35
      kwp: 6.5
    - name: "Garage West"
      azimuth: 270
      tilt: 20
      kwp: 4.0

# Your Home Assistant sensor mappings
input_sensors:
  battery_soc: sensor.inverter_battery_soc
  pv_power: sensor.inverter_pv_power
  load_power: sensor.inverter_load_power
  # Darkstar automatically aggregates all production sensors
  # if you use sensor.pv_total or similar integration.

# Nordpool price area
nordpool:
  price_area: "SE4"
```

### `secrets.yaml`

Credentials (never committed to git):

```yaml
home_assistant:
  url: "http://your-homeassistant:8123"
  token: "your-long-lived-access-token"

openrouter_api_key: "sk-or-v1-..."

notifications:
  discord_webhook_url: ""  # Optional fallback alerts
```

### Water Heater (Optional)

If you have a smart water heater, Darkstar can optimize its heating schedule. It controls the water heater thermostat temperature with set levels:

```yaml
# In config.yaml - Entity-Centric Configuration (ARC15)
water_heaters:
  - id: main_tank
    name: "Main Water Heater"
    enabled: true
    power_kw: 3.0
    min_kwh_per_day: 6.0
    max_hours_between_heating: 8
    water_min_spacing_hours: 4
    sensor: sensor.vvb_power
    type: binary                # or `thermal` for the tank-level estimator (see fork docs)
    target_entity: input_number.your_water_heater_temp   # per-heater control entity

# Temperature levels used when controlling the heater
executor:
  water_heater:
    temp_off: 40       # Idle (legionella-safe minimum)
    temp_normal: 60    # Normal scheduled heating
    temp_boost: 70     # Manual boost via Dashboard
    temp_max: 85       # Max temp for PV surplus dumping
```

| Setting       | Default | When Used                      |
| ------------- | ------- | ------------------------------ |
| `temp_off`    | 40°C    | Idle mode, no active heating   |
| `temp_normal` | 60°C    | Planner schedules heating      |
| `temp_boost`  | 70°C    | You press "Water Boost" button |
| `temp_max`    | 85°C    | Excess PV with full battery    |

### EV Charger (Optional)

Darkstar can optimize EV charging as a deferrable load:

```yaml
# Entity-Centric Configuration (ARC15)
ev_chargers:
  - id: tesla_model_3
    name: "Tesla Model 3"
    enabled: true
    max_power_kw: 11.0
    battery_capacity_kwh: 82.0
    sensor: sensor.tesla_power                    # charger power reading
    soc_sensor: sensor.tesla_battery_level        # EV battery SoC (%)
    plug_sensor: binary_sensor.tesla_plugged_in   # plug status
    switch_entity: switch.tesla_charger           # enable/disable smart charging
    type: binary                                  # ON/OFF control (variable power not yet implemented)
    nominal_power_kw: 11.0
    departure_time: ""                            # optional "HH:MM" ready-by time
```

### Multiple Devices

You can add multiple water heaters and EV chargers:

```yaml
water_heaters:
  - id: main_tank
    name: "Main Water Heater"
    enabled: true
    # ... settings
  - id: upstairs_tank
    name: "Upstairs Water Heater"
    enabled: true
    # ... settings

ev_chargers:
  - id: tesla
    name: "Tesla Model 3"
    enabled: true
    # ... settings
  - id: fiat
    name: "Fiat 500e"
    enabled: false  # Disabled, won't be used
    # ... settings
```


## Home Assistant Integration

Darkstar reads sensors and controls your inverter through Home Assistant:

**Required Sensors:**
- Battery SoC (%)
- Total PV production (kWh)
- Total load consumption (kWh)

**Controlled Entities:**
- Inverter work mode (export/zero-export)
- Battery charging current limits
- Grid charging switch
- Water heater temperature (optional)

## Dashboard

The web UI provides:

- **Live Schedule** — 48-hour visualization with charge/discharge/export slots
- **Forecasting** — Compare ML predictions vs. actuals
- **Manual Planning** — Override or extend the automated schedule
- **Settings** — Tune parameters without editing YAML files


## How It Works

1. **Forecasting** — Aurora ML predicts your home's energy patterns
2. **Strategy** — Context-aware adjustments (vacation mode, weather, etc.)
3. **Optimization** — Kepler solver generates optimal battery schedules
4. **Execution** — Native executor controls your inverter in real-time

The system re-optimizes every 30 minutes (configurable via `automation.schedule.every_minutes`) to adapt to changing prices and conditions.

---

## License

Licensed under the **[GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)**.
