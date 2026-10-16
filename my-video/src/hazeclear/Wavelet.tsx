import {
  Interactive,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Stage, Picture, clamp, colors } from "./Visuals";

const WaveletInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <Stage
      chapter="02 / WAVELET"
      title="Tách ảnh thành các dải tần"
      subtitle="DWT tách cấu trúc tổng thể khỏi biến thiên nhanh của ảnh."
      note="Haar DWT trên ảnh minh họa • Dải chi tiết được tăng tương phản để nhìn rõ"
    >
      <div
        style={{
          position: "absolute",
          left: 70,
          top: 284,
          width: 360,
          height: 270,
        }}
      >
        <Picture name="hazy" />
      </div>
      <div
        style={{
          position: "absolute",
          left: 454,
          top: 378,
          fontSize: 45,
          color: colors.red,
        }}
      >
        →
      </div>
      {["LL", "LH", "HL", "HH"].map((band, i) => (
        <div
          key={band}
          style={{
            position: "absolute",
            left: 550 + (i % 2) * 310,
            top: 224 + Math.floor(i / 2) * 203,
            width: 270,
            height: 165,
            opacity: interpolate(
              frame,
              [0.6 * fps + i * 0.35 * fps, 1.3 * fps + i * 0.35 * fps],
              [0, 1],
              clamp,
            ),
            translate: interpolate(
              frame,
              [0.6 * fps + i * 0.35 * fps, 1.3 * fps + i * 0.35 * fps],
              ["-40px 0px", "0px 0px"],
              clamp,
            ),
          }}
        >
          <Picture name={band} />
          <div
            style={{
              position: "absolute",
              left: 10,
              top: 10,
              background: i === 0 ? colors.red : colors.ink,
              color: "white",
              padding: "4px 12px",
              fontSize: 23,
              fontWeight: 700,
            }}
          >
            {band}
          </div>
          <div
            style={{
              position: "absolute",
              bottom: -27,
              fontSize: 19,
              color: i === 0 ? colors.red : colors.muted,
            }}
          >
            {i === 0 ? "Cấu trúc, độ sáng, tương phản" : "Biên và texture"}
          </div>
        </div>
      ))}
      <div
        style={{
          position: "absolute",
          left: 70,
          top: 585,
          width: 375,
          fontSize: 25,
          color: colors.muted,
        }}
      >
        Mỗi dải có kích thước
        <br />
        <b>H/2 × W/2</b>
      </div>
    </Stage>
  );
};
export const Wavelet = Interactive.withSchema({
  Component: WaveletInner,
  componentName: "HazeClearWavelet",
  schema: {},
  wrapInSequence: true,
  layout: "absolute-fill",
});
