"""Support for LED lights."""
from __future__ import annotations

from functools import partial
from typing import Any, cast

from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_EFFECT,
    ATTR_RGB_COLOR,
    ATTR_RGBW_COLOR,
    ATTR_TRANSITION,
    ATTR_RGBWW_COLOR,
    ColorMode,
    LightEntity,
    LightEntityFeature, ATTR_COLOR_TEMP_KELVIN,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import ATTR_COLOR_PRIMARY, ATTR_ON, ATTR_SEGMENT_ID, DOMAIN, LOGGER
from .coordinator import WLEDDataUpdateCoordinator
from .helpers import wled_exception_handler
from .models import WLEDEntity

PARALLEL_UPDATES = 1
ADDITIVE_BLENDING = 100


def cct_to_cw_ww(cct: int) -> tuple[int, int]:
    """Convert cct value (0-255) to cw/ww values (0-255)."""

    cw = min(int(cct * (100 + ADDITIVE_BLENDING) / 100), 255)
    ww = min(int((255 - cct) * (100 + ADDITIVE_BLENDING) / 100), 255)
    return cw, ww


def cw_ww_to_cct(cw: int, ww: int) -> int:
    """Convert cw/ww values (0-255) to cct value (0-255)."""
    if cw == 0 and ww == 0:
        return 127
    if cw == 255:
        cw = 255 * (100 + ADDITIVE_BLENDING) / 100 - ww
    elif ww == 255:
        ww = 255 * (100 + ADDITIVE_BLENDING) / 100 - cw
    cct = int(cw / (cw + ww) * 255)
    return cct


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up WLED light based on a config entry."""
    coordinator: WLEDDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]
    if coordinator.keep_main_light:
        async_add_entities([WLEDMainLight(coordinator=coordinator)])

    update_segments = partial(
        async_update_segments,
        coordinator,
        set(),
        async_add_entities,
    )

    coordinator.async_add_listener(update_segments)
    update_segments()


class WLEDMainLight(WLEDEntity, LightEntity):
    """Defines a WLED main light."""

    _attr_color_mode = ColorMode.BRIGHTNESS
    _attr_icon = "mdi:led-strip-variant"
    _attr_translation_key = "main"
    _attr_supported_features = LightEntityFeature.TRANSITION
    _attr_supported_color_modes = {ColorMode.BRIGHTNESS}

    def __init__(self, coordinator: WLEDDataUpdateCoordinator) -> None:
        """Initialize WLED main light."""
        super().__init__(coordinator=coordinator)
        self._attr_unique_id = coordinator.data.info.mac_address

    @property
    def brightness(self) -> int | None:
        """Return the brightness of this light between 1..255."""
        return self.coordinator.data.state.brightness

    @property
    def is_on(self) -> bool:
        """Return the state of the light."""
        return bool(self.coordinator.data.state.on)

    @property
    def available(self) -> bool:
        """Return if this main light is available or not."""
        return self.coordinator.has_main_light and super().available

    @wled_exception_handler
    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off the light."""
        transition = None
        if ATTR_TRANSITION in kwargs:
            # WLED uses 100ms per unit, so 10 = 1 second.
            transition = round(kwargs[ATTR_TRANSITION] * 10)

        await self.coordinator.wled.master(on=False, transition=transition)

    @wled_exception_handler
    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on the light."""
        transition = None
        if ATTR_TRANSITION in kwargs:
            # WLED uses 100ms per unit, so 10 = 1 second.
            transition = round(kwargs[ATTR_TRANSITION] * 10)

        await self.coordinator.wled.master(
            on=True, brightness=kwargs.get(ATTR_BRIGHTNESS), transition=transition
        )


class WLEDSegmentLight(WLEDEntity, LightEntity):
    """Defines a WLED light based on a segment."""

    _attr_supported_features = LightEntityFeature.EFFECT | LightEntityFeature.TRANSITION
    _attr_icon = "mdi:led-strip-variant"

    def __init__(
        self,
        coordinator: WLEDDataUpdateCoordinator,
        segment: int,
    ) -> None:
        """Initialize WLED segment light."""
        super().__init__(coordinator=coordinator)
        self._rgbw = coordinator.data.info.leds.rgbw
        self._wv = coordinator.data.info.leds.wv
        self._cct = coordinator.data.info.leds.cct
        self._segment = segment

        # Segment 0 uses a simpler name, which is more natural for when using
        # a single segment / using WLED with one big LED strip.
        if segment == 0:
            self._attr_name = None
        else:
            self._attr_name = f"Segment {segment}"

        self._attr_unique_id = (
            f"{self.coordinator.data.info.mac_address}_{self._segment}"
        )

        self._attr_color_mode = ColorMode.RGB
        self._attr_supported_color_modes = {ColorMode.RGB}
        if self._rgbw and self._cct:
            self._attr_color_mode = ColorMode.RGBWW
            self._attr_supported_color_modes = {ColorMode.RGB, ColorMode.RGBWW, ColorMode.COLOR_TEMP}
            self._attr_min_color_temp_kelvin = 2700
            self._attr_max_color_temp_kelvin = 6500
        elif self._rgbw and self._wv:
            self._attr_color_mode = ColorMode.RGBW
            self._attr_supported_color_modes = {ColorMode.RGBW}

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        try:
            self.coordinator.data.state.segments[self._segment]
        except IndexError:
            return False

        return super().available

    @property
    def rgb_color(self) -> tuple[int, int, int] | None:
        """Return the color value."""
        return self.coordinator.data.state.segments[self._segment].color_primary[:3]

    @property
    def rgbw_color(self) -> tuple[int, int, int, int] | None:
        """Return the color value."""
        return cast(
            tuple[int, int, int, int],
            self.coordinator.data.state.segments[self._segment].color_primary,
        )

    @property
    def rgbww_color(self) -> tuple[int, int, int, int, int] | None:
        """Return the color value."""
        r, g, b, w = self.coordinator.data.state.segments[self._segment].color_primary
        cw, ww = cct_to_cw_ww(self.coordinator.data.state.segments[self._segment].cct)
        cw_w = int(cw / 255 * w / 255 * 255)
        ww_w = int(ww / 255 * w / 255 * 255)
        return cast(
            tuple[int, int, int, int, int],
            (r, g, b, cw_w, ww_w)
        )

    @property
    def effect(self) -> str | None:
        """Return the current effect of the light."""
        return self.coordinator.data.state.segments[self._segment].effect.name

    @property
    def brightness(self) -> int | None:
        """Return the brightness of this light between 1..255."""
        state = self.coordinator.data.state

        # If this is the one and only segment, calculate brightness based
        # on the main and segment brightness
        if not self.coordinator.has_main_light:
            return int(
                (state.segments[self._segment].brightness * state.brightness) / 255
            )

        return state.segments[self._segment].brightness

    @property
    def color_temp_kelvin(self) -> int | None:
        """Return the color temperature."""
        # 2700K (cct 0) to 6500K (cct 255)
        return int((self.coordinator.data.state.segments[self._segment].cct / 255) * (6500 - 2700) + 2700)

    @property
    def effect_list(self) -> list[str]:
        """Return the list of supported effects."""
        return [effect.name for effect in self.coordinator.data.effects]

    @property
    def is_on(self) -> bool:
        """Return the state of the light."""
        state = self.coordinator.data.state

        # If there is no main, we take the main state into account
        # on the segment level.
        if not self.coordinator.has_main_light and not state.on:
            return False

        return bool(state.segments[self._segment].on)

    @wled_exception_handler
    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off the light."""
        transition = None
        if ATTR_TRANSITION in kwargs:
            # WLED uses 100ms per unit, so 10 = 1 second.
            transition = round(kwargs[ATTR_TRANSITION] * 10)

        # If there is no main control, and only 1 segment, handle the main
        if not self.coordinator.has_main_light:
            await self.coordinator.wled.master(on=False, transition=transition)
            return

        await self.coordinator.wled.segment(
            segment_id=self._segment, on=False, transition=transition
        )

    @wled_exception_handler
    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on the light."""
        LOGGER.error(kwargs)
        data: dict[str, Any] = {
            ATTR_ON: True,
            ATTR_SEGMENT_ID: self._segment,
        }

        if ATTR_RGB_COLOR in kwargs:
            data[ATTR_COLOR_PRIMARY] = kwargs[ATTR_RGB_COLOR]

        if ATTR_RGBW_COLOR in kwargs:
            data[ATTR_COLOR_PRIMARY] = kwargs[ATTR_RGBW_COLOR]

        if ATTR_TRANSITION in kwargs:
            # WLED uses 100ms per unit, so 10 = 1 second.
            data[ATTR_TRANSITION] = round(kwargs[ATTR_TRANSITION] * 10)

        if ATTR_BRIGHTNESS in kwargs:
            data[ATTR_BRIGHTNESS] = kwargs[ATTR_BRIGHTNESS]

        if ATTR_EFFECT in kwargs:
            data[ATTR_EFFECT] = kwargs[ATTR_EFFECT]

        if ATTR_RGBWW_COLOR in kwargs:
            r, g, b, cw_w, ww_w = kwargs[ATTR_RGBWW_COLOR]
            w = max(cw_w, ww_w)
            if w == 0:
                w = 1
            data[ATTR_COLOR_PRIMARY] = (r, g, b, w)
            cw = int(cw_w / w * 255)
            ww = int(ww_w / w * 255)
            data["cct"] = cw_ww_to_cct(cw, ww)

        if ATTR_COLOR_TEMP_KELVIN in kwargs:
            # 2700K (cct 0) to 6500K (cct 255)
            data["cct"] = int((kwargs[ATTR_COLOR_TEMP_KELVIN] - 2700) / (6500 - 2700) * 255)
            data[ATTR_COLOR_PRIMARY] = (0, 0, 0, 255)

        # If there is no main control, and only 1 segment, handle the main
        if not self.coordinator.has_main_light:
            main_data = {ATTR_ON: True}
            if ATTR_BRIGHTNESS in data:
                main_data[ATTR_BRIGHTNESS] = data[ATTR_BRIGHTNESS]
                data[ATTR_BRIGHTNESS] = 255

            if ATTR_TRANSITION in data:
                main_data[ATTR_TRANSITION] = data[ATTR_TRANSITION]
                del data[ATTR_TRANSITION]

            await self.coordinator.wled.segment(**data)
            await self.coordinator.wled.master(**main_data)
            return

        await self.coordinator.wled.segment(**data)


@callback
def async_update_segments(
    coordinator: WLEDDataUpdateCoordinator,
    current_ids: set[int],
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Update segments."""
    segment_ids = {light.segment_id for light in coordinator.data.state.segments}
    new_entities: list[WLEDMainLight | WLEDSegmentLight] = []

    # More than 1 segment now? No main? Add main controls
    if not coordinator.keep_main_light and (
        len(current_ids) < 2 and len(segment_ids) > 1
    ):
        new_entities.append(WLEDMainLight(coordinator))

    # Process new segments, add them to Home Assistant
    for segment_id in segment_ids - current_ids:
        current_ids.add(segment_id)
        new_entities.append(WLEDSegmentLight(coordinator, segment_id))

    async_add_entities(new_entities)
