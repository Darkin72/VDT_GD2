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
      chapter="09 / KẾT QUẢ"
      title="Kết quả và hướng tiếp theo"
      subtitle="DSRB tăng FPS nhưng giảm PSNR."
      note="Thử nghiệm tiếp theo: số kênh, số khối, kết nối tắt và chưng cất mô hình."
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
          KẾT QUẢ
        </div>
        <div style={{ fontSize: 54, color: colors.red, fontWeight: 700, marginTop: 12 }}>
          37,26 FPS
        </div>
        <div style={{ fontSize: 27, marginTop: 8 }}>
          DSRB trên SOTS-OTS
        </div>
        <div style={{ fontSize: 23, color: colors.muted, marginTop: 14, lineHeight: 1.4 }}>
          PSNR: 31,18 dB (−1,17 dB).
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
          HƯỚNG TIẾP THEO
        </div>
        <div style={{ fontSize: 29, lineHeight: 1.42, marginTop: 18 }}>
          Giữ mức tăng tốc.
          <br />Giảm suy hao PSNR xuống ≤ 0,3 dB.
          <br />Đánh giá trên video và tác vụ nhận dạng.
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
        Mục tiêu ứng dụng: tiền xử lý ảnh cho camera và hệ thống thị giác máy tính.
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
