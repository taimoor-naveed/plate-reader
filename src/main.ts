import './web/ui.css'
import { installViewportHeightVar } from './web/viewport'
import './web/app'
import { registerServiceWorker } from './web/sw-register'

installViewportHeightVar()
registerServiceWorker()
