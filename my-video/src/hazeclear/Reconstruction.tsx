import {
  Interactive,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Stage, Picture, Pipeline, clamp, colors } from "./Visuals";

const ReconstructionInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <Stage
      chapter="04 / IWT"
      title="Tái dựng ảnh sau phục hồi"
      subtitle="Wavelet ngược đưa biểu diễn trở lại không gian ảnh RGB."
      note="Ảnh thô là mô phỏng làm mờ ảnh paper để giải thích trình tự • Không phải đầu ra LFRN đã đo"
    >
      <div
        style={{
          position: "absolute",
          left: 72,
          top: 270,
          width: 370,
          fontSize: 36,
          lineHeight: 1.25,
        }}
      >
        Tần số thấp
        <br />
        <b style={{ color: colors.red }}>đã phục hồi</b>
        <br />
        <span style={{ fontSize: 27, color: colors.muted }}>
          cùng các dải chi tiết
        </span>
      </div>
      <div
        style={{
          position: "absolute",
          left: 450,
          top: 330,
          fontSize: 44,
          color: colors.red,
        }}
      >
        IWT →
      </div>
      <div
        style={{
          position: "absolute",
          left: 665,
          top: 228,
          width: 510,
          height: 320,
          opacity: interpolate(frame, [fps, 2 * fps], [0, 1], clamp),
          scale: interpolate(frame, [fps, 3 * fps], [0.85, 1], clamp),
        }}
      >
        <Picture name="clear" style={{ filter: "blur(3px)" }} />
        <div
          style={{
            position: "absolute",
            bottom: 14,
            left: 17,
            color: "white",
            background: "#183544DD",
            padding: "6px 15px",
            fontSize: 22,
          }}
        >
          Ảnh khử sương thô (mô phỏng)
        </div>
      </div>
      <div style={{ position: "absolute", left: 90, top: 595 }}>
        <Pipeline active={3} />
      </div>
    </Stage>
  );
};
export const Reconstruction = Interactive.withSchema({
  Component: ReconstructionInner,
  componentName: "HazeClearReconstruction",
  schema: {},
  wrapInSequence: true,
  layout: "absolute-fill",
});
