# Remotion video

## Viettel HazeClear

Composition `HazeClearExplainer` minh họa cơ chế khử sương bằng Wavelet,
Mamba/LFRN, IWT và cấu hình tối ưu thay DEN bằng DSRB. Thời lượng 74 giây,
tiếng Việt, 1280 x 720, 30 fps.
Animation chú vịt bên dưới vẫn được giữ nguyên.

Mở `npm run dev`, chọn `HazeClearExplainer`. Xem kịch bản, nguồn hình và lệnh
xuất MP4 trong [HAZECLEAR.md](HAZECLEAR.md).

Kiểm tra mã và các khung hình:

```console
npm run lint
node scripts/check-hazeclear.cjs
```

<p align="center">
  <a href="https://github.com/remotion-dev/logo">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/remotion-dev/logo/raw/main/animated-logo-banner-dark.apng">
      <img alt="Animated Remotion Logo" src="https://github.com/remotion-dev/logo/raw/main/animated-logo-banner-light.gif">
    </picture>
  </a>
</p>

Welcome to your Remotion project!

## Commands

**Install Dependencies**

```console
npm i --loglevel=error
```

**Start Preview**

```console
npm run dev
```

**Render video**

```console
npx remotion render MyComp out/duck-bike.mp4
```

## Duck bicycle animation

Composition `MyComp` trong `src/Composition.tsx` là animation chú vịt đội mũ đạp xe
qua đường. Bánh xe, pedal, chân vịt, thân vịt, mây và vạch đường đều chuyển động theo
timeline. Bạn có thể đổi nội dung và màu nhấn qua `defaultProps`:

```tsx
defaultProps={{
  title: "VỊT ĐẠP XE",
  subtitle: "Một chuyến đi thật vui",
  accent: "#f7c948",
}}
```

Mở preview bằng `npm run dev`, chọn `MyComp`, hoặc render trực tiếp bằng lệnh ở trên.

**Upgrade Remotion**

```console
npx remotion upgrade
```

## Docs

Get started with Remotion by reading the [fundamentals page](https://www.remotion.dev/docs/the-fundamentals).

## Help

We provide help on our [Discord server](https://discord.gg/6VzzNDwUwV).

## Issues

Found an issue with Remotion? [File an issue here](https://github.com/remotion-dev/remotion/issues/new).

## License

Note that for some entities a company license is needed. [Read the terms here](https://github.com/remotion-dev/remotion/blob/main/LICENSE.md).
