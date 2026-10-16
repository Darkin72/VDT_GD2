import {
  Interactive,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Stage, Picture, clamp, colors } from "./Visuals";

const MambaInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = Math.max(0, frame / fps - 1);
  const direction = Math.min(3, Math.floor(t / 2));
  const index = Math.min(47, Math.floor(((t - direction * 2) / 2) * 48));
  // Four scan orders illustrate SS2D without claiming to visualize learned attention.
  const cell =
    direction === 0
      ? index
      : direction === 1
        ? 47 - index
        : direction === 2
          ? (index % 6) * 8 + Math.floor(index / 6)
          : (5 - (index % 6)) * 8 + 7 - Math.floor(index / 6);
  return (
    <Stage
      chapter="03 / LFRN + MAMBA"
      title="Kết nối thông tin trên toàn ảnh"
      subtitle="Quét đặc trưng theo bốn hướng và hợp nhất ngữ cảnh không gian."
      note="Sơ đồ minh họa SS2D • Không phải attention map hoặc đặc trưng xuất từ mô hình"
    >
      <div
        style={{
          position: "absolute",
          left: 68,
          top: 239,
          width: 640,
          height: 360,
          overflow: "hidden",
          borderRadius: 7,
        }}
      >
        <Picture name="LL" />
        {Array.from({ length: 48 }, (_, i) => (
          <div
            key={i}
            style={{
              position: "absolute",
              left: (i % 8) * 80,
              top: Math.floor(i / 8) * 60,
              width: 80,
              height: 60,
              border: "1px solid #FFFFFF66",
              boxSizing: "border-box",
              background: i === cell ? colors.red + "BB" : "transparent",
            }}
          />
        ))}
        <div
          style={{
            position: "absolute",
            inset: 0,
            border: `3px solid ${colors.red}`,
            opacity: interpolate(frame, [9 * fps, 10 * fps], [0, 1], clamp),
          }}
        />
      </div>
      <div style={{ position: "absolute", left: 780, top: 250, width: 430 }}>
        <div style={{ fontSize: 88, color: colors.red, fontWeight: 700 }}>
          {["→", "←", "↓", "↑"][direction]}
        </div>
        <div style={{ fontSize: 38, lineHeight: 1.18 }}>
          Chọn lọc thông tin
          <br />
          theo đầu vào
        </div>
        <div
          style={{
            marginTop: 25,
            fontSize: 26,
            color: colors.muted,
            lineHeight: 1.4,
          }}
        >
          Truyền trạng thái qua các vị trí.
          <br />
          Khôi phục cấu trúc toàn cục.
        </div>
      </div>
      <div
        style={{
          position: "absolute",
          left: 70,
          top: 625,
          fontSize: 25,
          color: colors.red,
          fontWeight: 700,
          opacity: interpolate(frame, [8.5 * fps, 9.5 * fps], [0, 1], clamp),
        }}
      >
        Hợp nhất bốn hướng trong mạng phục hồi tần số thấp
      </div>
    </Stage>
  );
};
export const Mamba = Interactive.withSchema({
  Component: MambaInner,
  componentName: "HazeClearMamba",
  schema: {},
  wrapInSequence: true,
  layout: "absolute-fill",
});
