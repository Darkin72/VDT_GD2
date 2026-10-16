import {
  Interactive,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Stage, Pipeline, clamp, colors } from "./Visuals";

const ClosingInner = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <Stage
      chapter="07 / KẾT QUẢ"
      title="Tăng tốc đã rõ, chất lượng cần phục hồi"
      subtitle="DSRB mở ra hướng triển khai nhẹ hơn nhưng chưa thể thay DEN hoàn toàn."
      note="Bước tiếp theo: tối ưu số kênh, số block, residual hoặc distillation; sau đó kiểm thử video và mAP/recall"
    >
      <div style={{ position: "absolute", left: 62, top: 220 }}>
        <Pipeline active={Math.min(5, Math.floor(frame / (0.8 * fps)))} />
      </div>
      <div
        style={{
          position: "absolute",
          left: 72,
          top: 350,
          width: 515,
          opacity: interpolate(frame, [2 * fps, 3 * fps], [0, 1], clamp),
        }}
      >
        <div style={{ fontSize: 26, color: colors.red, fontWeight: 700 }}>
          KẾT QUẢ HIỆN TẠI
        </div>
        <div style={{ fontSize: 54, color: colors.red, fontWeight: 700, marginTop: 12 }}>
          37,26 FPS
        </div>
        <div style={{ fontSize: 27, marginTop: 8 }}>
          DSRB trên SOTS-OTS
        </div>
        <div style={{ fontSize: 23, color: colors.muted, marginTop: 14, lineHeight: 1.4 }}>
          PSNR 31,18 dB, giảm 1,17 dB so với baseline.
        </div>
      </div>
      <div
        style={{
          position: "absolute",
          left: 695,
          top: 350,
          width: 505,
          opacity: interpolate(frame, [3 * fps, 4 * fps], [0, 1], clamp),
        }}
      >
        <div style={{ fontSize: 26, color: colors.teal, fontWeight: 700 }}>
          HƯỚNG HOÀN THIỆN
        </div>
        <div style={{ fontSize: 29, lineHeight: 1.42, marginTop: 18 }}>
          Giữ mức tăng tốc.
          <br />Giảm suy hao PSNR xuống ≤ 0,3 dB.
          <br />Kiểm thử camera, video và tác vụ nhận dạng.
        </div>
      </div>
      <div
        style={{
          position: "absolute",
          left: 75,
          top: 610,
          right: 70,
          fontSize: 25,
          color: colors.ink,
          fontWeight: 700,
        }}
      >
        Haze Clear hướng tới mô-đun tiền xử lý ảnh cho camera và hệ thống thị giác máy tính.
      </div>
    </Stage>
  );
};
export const Closing = Interactive.withSchema({
  Component: ClosingInner,
  componentName: "HazeClearClosing",
  schema: {},
  wrapInSequence: true,
  layout: "absolute-fill",
});
