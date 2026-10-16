import {
  Interactive,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Stage, clamp, colors } from "./Visuals";

const profiling = [
  { label: "DWT", value: 0.0327, display: "0,03" },
  { label: "LFRN", value: 9.0026, display: "9,00" },
  { label: "IWT", value: 0.7997, display: "0,80" },
  { label: "DEN", value: 24.9351, display: "24,94" },
];

const DetailInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <Stage
      chapter="06 / ĐO THỜI GIAN DEN"
      title="DEN có thời gian xử lý lớn nhất"
      subtitle="Thời gian GPU của từng khối được đo trên RESIDE-6K."
      note="Điều kiện: 1.000 ảnh • A100 80 GB • FP32 • lô 100 ảnh • cạnh tối đa 512 px"
    >
      <div
        style={{
          position: "absolute",
          left: 70,
          top: 230,
          width: 725,
        }}
      >
        {profiling.map((item, index) => {
          const width = interpolate(
            frame,
            [0.8 * fps + index * 0.25 * fps, 2.2 * fps + index * 0.25 * fps],
            [0, (item.value / 27) * 590],
            clamp,
          );
          const highlight = item.label === "DEN";
          return (
            <div
              key={item.label}
              style={{
                display: "grid",
                gridTemplateColumns: "86px 1fr 86px",
                alignItems: "center",
                gap: 15,
                height: 76,
                fontSize: 24,
              }}
            >
              <div style={{ fontWeight: highlight ? 700 : 500 }}>{item.label}</div>
              <div style={{ height: 34, background: "#DCE5E6", borderRadius: 4 }}>
                <div
                  style={{
                    width,
                    height: "100%",
                    borderRadius: 4,
                    background: highlight ? colors.red : colors.ink,
                  }}
                />
              </div>
              <div style={{ color: highlight ? colors.red : colors.muted }}>
                {item.display} ms
              </div>
            </div>
          );
        })}
      </div>
      <div
        style={{
          position: "absolute",
          left: 855,
          top: 238,
          width: 355,
          opacity: interpolate(frame, [2.4 * fps, 3.2 * fps], [0, 1], clamp),
        }}
      >
        <div style={{ fontSize: 68, color: colors.red, fontWeight: 700 }}>
          71,71%
        </div>
        <div style={{ fontSize: 27, lineHeight: 1.35, marginTop: 9 }}>
          thời gian đo được thuộc về DEN.
        </div>
        <div
          style={{
            marginTop: 32,
            padding: "18px 20px",
            borderLeft: `5px solid ${colors.red}`,
            background: "#FFFFFFAA",
            fontSize: 25,
            lineHeight: 1.35,
          }}
        >
          DEN dùng tích chập dày và xử lý miền Fourier.
          <br />
          DEN được chọn để tối ưu.
        </div>
      </div>
    </Stage>
  );
};
export const Detail = Interactive.withSchema({
  Component: DetailInner,
  componentName: "HazeClearDetail",
  schema: {},
  wrapInSequence: true,
  layout: "absolute-fill",
});
