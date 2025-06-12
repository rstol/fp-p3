# Basketball Analytics Frontend

A React 19 application providing interactive visualizations for NBA player tracking data analysis.

## Tech Stack

- **React 19** with TypeScript for type safety
- **Tailwind CSS** for utility-first styling
- **shadcn/ui** for accessible component library
- **D3.js** for custom data visualizations
- **React Router** for client-side navigation

## Quick Start

### Installation

```bash
npm install --force
```

> **Note**: `--force` is required due to React 19 compatibility issues with the `emblor` package. This is a temporary issue.

### Development

```bash
npm run dev
```

Application runs at `http://localhost:3000` with hot module replacement.

### Docker Development

```bash
# Start frontend only
docker compose up -d frontend

# With rebuild (after package changes)
docker compose up frontend --build
```

## Project Structure

```
.
├── app
│   ├── app.css
│   ├── components      # Reusable UI components
│   ├── lib
│   ├── root.tsx
│   ├── routes
│   ├── routes.ts
│   ├── setupTests.ts
│   └── types
├── components.json
├── Dockerfile
├── globals.d.ts
├── package-lock.json
├── package.json
├── public
│   ├── favicon.ico
│   ├── spurs.gif
│   └── videos
├── react-router.config.ts
├── README.md
├── tsconfig.json
└── vite.config.ts
```

## Styling Architecture

### Tailwind CSS
Utility-first CSS framework configured for rapid development:

```jsx
<div className="flex items-center justify-between p-4 bg-slate-100 rounded-lg">
  {/* Component content */}
</div>
```

### shadcn/ui Components
Pre-built accessible components. Add new components:

```bash
npx shadcn-ui@latest add button
npx shadcn-ui@latest add dialog
```

### Common Issues

**Hot reload not working:**
```bash
# Clear cache and restart
rm -rf node_modules/.cache
npm run dev
```

**Build failures:**
```bash
# Check TypeScript errors
npm run type-check

# Verify all dependencies
npm install --force
```

**API connection issues:**
- Verify backend is running on correct port
- Check CORS configuration
- Confirm API endpoint URLs in environment variables