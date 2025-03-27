# Welcome to React Router!

A modern, production-ready template for building full-stack React applications using React Router.

[![Open in StackBlitz](https://developer.stackblitz.com/img/open_in_stackblitz.svg)](https://stackblitz.com/github/remix-run/react-router-templates/tree/main/default)

## Features

- 🚀 Server-side rendering
- ⚡️ Hot Module Replacement (HMR)
- 📦 Asset bundling and optimization
- 🔄 Data loading and mutations
- 🔒 TypeScript by default
- 🎉 TailwindCSS for styling
- 📖 [React Router docs](https://reactrouter.com/)

## Getting Started

### Installation

Install the dependencies:

```bash
npm install
npm prepare # Sets up husky for pre-commit formatting
```

### Development

Start the development server with HMR:

```bash
npm run dev
```

Your application will be available at `http://localhost:5173`.

#### Development with Docker

You can also run the frontend using Docker Compose:

```bash
docker compose up -d frontend
```

Note: If you add new packages to the project, you'll need to rebuild the container:

```bash
docker compose up frontend --build
```

## Building for Production

Create a production build:

```bash
npm run build
```

## Deployment

### Docker Deployment

To build and run using Docker:

```bash
docker build -t my-app .

# Run the container
docker run -p 3000:3000 my-app
```

The containerized application can be deployed to any platform that supports Docker, including:

- AWS ECS
- Google Cloud Run
- Azure Container Apps
- Digital Ocean App Platform
- Fly.io
- Railway

### DIY Deployment

If you're familiar with deploying Node applications, the built-in app server is production-ready.

Make sure to deploy the output of `npm run build`

```
├── package.json
├── package-lock.json (or pnpm-lock.yaml, or bun.lockb)
├── build/
│   ├── client/    # Static assets
│   └── server/    # Server-side code
```

## Styling

This template comes with several styling solutions pre-configured:

### Tailwind CSS

[Tailwind CSS](https://tailwindcss.com/) is configured out of the box for utility-first styling. You can start using Tailwind classes immediately in your components:

```jsx
<div className="flex items-center justify-between p-4">
  {/* Your component content */}
</div>
```

### shadcn/ui Components

This project includes [shadcn/ui](https://ui.shadcn.com/) for beautiful, accessible, and customizable components. To add new components:

1. Use the CLI to add components:

```bash
npx shadcn-ui@latest add button
```

2. Import and use components in your React files:

```jsx
import { Button } from '@/components/ui/button';

export default function Page() {
  return <Button>Click me</Button>;
}
```

All shadcn/ui components are built with Tailwind CSS and can be customized through the `tailwind.config.js` file.

---

Built with ❤️ using React Router.
