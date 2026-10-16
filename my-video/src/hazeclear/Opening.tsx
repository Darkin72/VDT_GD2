import {
  Interactive,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Stage, Wipe, clamp, colors } from "./Visuals";

const OpeningInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <Stage
      chapter="01 / BÀI TOÁN"
      title="Một ảnh rõ hơn bắt đầu từ đâu?"
      subtitle="Sương làm mất tương phản, màu sắc và chi tiết."
      note="Ảnh trước/sau trích từ WDMamba, Hình 9 • Chuyển cảnh là minh họa, không chạy suy luận trực tiếp"
    >
      <div style={{ position: "absolute", left: 58, top: 228 }}>
        <Wipe width={750} height={390} start={1.5} end={6} />
      </div>
      <div
        style={{
          position: "absolute",
          left: 865,
          top: 277,
          width: 350,
          opacity: interpolate(frame, [0.5 * fps, 1.2 * fps], [0, 1], clamp),
        }}
      >
        <div style={{ fontSize: 26, color: colors.red, fontWeight: 700 }}>
          Ý TƯỞNG GIẢI PHÁP
        </div>
        <div style={{ fontSize: 43, lineHeight: 1.23, marginTop: 24 }}>
          Khôi phục cấu trúc.
          <br />
          Sau đó làm rõ chi tiết.
        </div>
        <div style={{ fontSize: 23, color: colors.muted, marginTop: 29 }}>
          Wavelet + Mamba + DSRB
        </div>
      </div>
    </Stage>
  );
};
export const Opening = Interactive.withSchema({
  Component: OpeningInner,
  componentName: "HazeClearOpening",
  schema: {},
  wrapInSequence: true,
  layout: "absolute-fill",
});
