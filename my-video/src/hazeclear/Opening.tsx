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
      title="Bài toán khử sương ảnh"
      subtitle="Sương làm giảm tương phản, sai lệch màu và che khuất chi tiết."
      showNote={false}
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
          QUY TRÌNH
        </div>
        <div style={{ fontSize: 43, lineHeight: 1.23, marginTop: 24 }}>
          Phục hồi cấu trúc.
          <br />
          Tinh chỉnh chi tiết.
        </div>
        <div style={{ fontSize: 23, color: colors.muted, marginTop: 29 }}>
          Wavelet • Mamba • DSRB
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
