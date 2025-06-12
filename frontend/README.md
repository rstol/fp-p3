# Welcome to the frontend

A React applications using React Router.

## Getting Started

### Installation

Install the dependencies:

```bash
npm install --force
```

The force is needed unfortunatetly since we are using the latest react.js version 19.x but the `emblor` package requirements are "incompatible" with react-19. 

### Development

Start the development server with HMR:

```bash
npm run dev
```

Your application will be available at `http://localhost:3000`.

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

## Styling

This template comes with several styling solutions pre-configured:

### Tailwind CSS

[Tailwind CSS](https://tailwindcss.com/) is configured out of the box for utility-first styling. You can start using Tailwind classes immediately in your components:

```jsx
<div className="flex items-center justify-between p-4">{/* Your component content */}</div>
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